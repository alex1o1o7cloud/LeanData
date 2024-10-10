import Mathlib

namespace ball_box_arrangement_l3359_335917

/-- The number of ways to arrange n balls in n boxes with exactly k balls in their corresponding boxes. -/
def arrangeWithExactMatches (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of derangements of n objects. -/
def derangement (n : ℕ) : ℕ :=
  sorry

theorem ball_box_arrangement :
  arrangeWithExactMatches 5 2 = 20 :=
sorry

end ball_box_arrangement_l3359_335917


namespace vector_equality_implies_x_coordinate_l3359_335905

/-- Given vectors a and b in ℝ², if |a + b| = |a - b|, then the x-coordinate of b is 1. -/
theorem vector_equality_implies_x_coordinate (a b : ℝ × ℝ) 
  (h : a = (-2, 1)) (h' : b.2 = 2) :
  ‖a + b‖ = ‖a - b‖ → b.1 = 1 := by
  sorry

#check vector_equality_implies_x_coordinate

end vector_equality_implies_x_coordinate_l3359_335905


namespace coin_triangle_border_mass_l3359_335990

/-- A configuration of coins in a triangular arrangement -/
structure CoinTriangle where
  total_coins : ℕ
  border_coins : ℕ
  trio_mass : ℝ

/-- The property that the total mass of border coins is a multiple of the trio mass -/
def border_mass_property (ct : CoinTriangle) : Prop :=
  ∃ k : ℕ, (ct.border_coins : ℝ) * ct.trio_mass / 3 = k * ct.trio_mass

/-- The theorem stating the total mass of border coins in the specific configuration -/
theorem coin_triangle_border_mass (ct : CoinTriangle) 
  (h1 : ct.total_coins = 28)
  (h2 : ct.border_coins = 18)
  (h3 : ct.trio_mass = 10) :
  (ct.border_coins : ℝ) * ct.trio_mass / 3 = 60 :=
sorry

end coin_triangle_border_mass_l3359_335990


namespace sum_of_fractions_and_decimal_l3359_335906

theorem sum_of_fractions_and_decimal : 
  (1 : ℚ) / 3 + 5 / 24 + (816 : ℚ) / 100 + 1 / 8 = 5296 / 600 := by
  sorry

end sum_of_fractions_and_decimal_l3359_335906


namespace min_split_links_for_all_weights_l3359_335937

/-- Represents a chain of links -/
structure Chain where
  totalLinks : Nat
  linkWeight : Nat

/-- Represents the result of splitting a chain -/
structure SplitChain where
  originalChain : Chain
  splitLinks : Nat

/-- Checks if all weights from 1 to the total weight can be assembled -/
def canAssembleAllWeights (sc : SplitChain) : Prop :=
  ∀ w : Nat, 1 ≤ w ∧ w ≤ sc.originalChain.totalLinks → 
    ∃ (subset : Finset Nat), subset.card ≤ sc.splitLinks + 1 ∧ 
      (subset.sum (λ i => sc.originalChain.linkWeight)) = w

/-- The main theorem -/
theorem min_split_links_for_all_weights 
  (c : Chain) 
  (h1 : c.totalLinks = 60) 
  (h2 : c.linkWeight = 1) :
  (∃ (k : Nat), k = 3 ∧ 
    canAssembleAllWeights ⟨c, k⟩ ∧
    (∀ (m : Nat), m < k → ¬canAssembleAllWeights ⟨c, m⟩)) :=
  sorry

end min_split_links_for_all_weights_l3359_335937


namespace cubic_equation_solution_l3359_335993

theorem cubic_equation_solution :
  let f (x : ℂ) := (x - 2)^3 + (x - 6)^3
  ∃ (s : Finset ℂ), s.card = 3 ∧ 
    (∀ x ∈ s, f x = 0) ∧
    (∀ x, f x = 0 → x ∈ s) ∧
    (4 ∈ s) ∧ 
    (4 + 2 * Complex.I * Real.sqrt 3 ∈ s) ∧ 
    (4 - 2 * Complex.I * Real.sqrt 3 ∈ s) :=
by
  sorry

end cubic_equation_solution_l3359_335993


namespace sum_of_prime_factors_of_2018_l3359_335913

theorem sum_of_prime_factors_of_2018 :
  ∀ p q : ℕ, 
  Prime p → Prime q → p * q = 2018 → p + q = 1011 := by
sorry

end sum_of_prime_factors_of_2018_l3359_335913


namespace max_min_values_of_f_l3359_335946

def f (x : ℝ) := 1 + x - x^2

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 4, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 4, f x = max) ∧
    (∀ x ∈ Set.Icc (-2) 4, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2) 4, f x = min) ∧
    max = 5/4 ∧ min = -11 :=
by sorry

end max_min_values_of_f_l3359_335946


namespace inscribed_squares_ratio_l3359_335960

/-- Given a right triangle with sides 5, 12, and 13 (13 being the hypotenuse),
    a square of side length x inscribed with one side along the leg of length 12,
    and another square of side length y inscribed with one side along the hypotenuse,
    the ratio of x to y is 12/13. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 → y > 0 →
  x^2 + x^2 = 5 * x →
  y^2 + y^2 = 13 * y →
  x / y = 12 / 13 := by
sorry

end inscribed_squares_ratio_l3359_335960


namespace calculation_proof_l3359_335947

theorem calculation_proof :
  (- (1 : ℤ) ^ 2023 + 8 * (-(1/2 : ℚ))^3 + |(-3 : ℤ)| = 1) ∧
  ((-25 : ℤ) * (3/2 : ℚ) - (-25 : ℤ) * (5/8 : ℚ) + (-25 : ℤ) / 8 = -25) := by
  sorry

end calculation_proof_l3359_335947


namespace no_two_cubes_between_squares_l3359_335954

theorem no_two_cubes_between_squares : ¬ ∃ (n a b : ℤ), n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n+1)^2 := by
  sorry

end no_two_cubes_between_squares_l3359_335954


namespace monica_reading_ratio_l3359_335974

/-- The number of books Monica read last year -/
def last_year : ℕ := 16

/-- The number of books Monica will read next year -/
def next_year : ℕ := 69

/-- The number of books Monica read this year -/
def this_year : ℕ := last_year * 2

theorem monica_reading_ratio :
  (this_year / last_year : ℚ) = 2 := by
  sorry

end monica_reading_ratio_l3359_335974


namespace complex_sum_magnitude_l3359_335972

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 3) : 
  Complex.abs (a + b + c) = 1 := by
sorry

end complex_sum_magnitude_l3359_335972


namespace max_element_of_A_l3359_335961

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) := Real.log x / Real.log 10

-- Define the set A
def A (x y : ℝ) : Set ℝ := {log x, log y, log (x + y/x)}

-- Define the theorem
theorem max_element_of_A :
  ∀ x y : ℝ, x > 0 → y > 0 → {0, 1} ⊆ A x y →
  ∃ z ∈ A x y, ∀ w ∈ A x y, w ≤ z ∧ z = log 11 :=
sorry

end max_element_of_A_l3359_335961


namespace prob_at_least_one_multiple_of_four_l3359_335929

/-- The number of integers from 1 to 60 inclusive -/
def total_numbers : ℕ := 60

/-- The number of multiples of 4 from 1 to 60 inclusive -/
def multiples_of_four : ℕ := 15

/-- The probability of choosing a number that is not a multiple of 4 -/
def prob_not_multiple_of_four : ℚ := (total_numbers - multiples_of_four) / total_numbers

theorem prob_at_least_one_multiple_of_four :
  1 - prob_not_multiple_of_four ^ 2 = 7 / 16 := by sorry

end prob_at_least_one_multiple_of_four_l3359_335929


namespace evaluate_expression_l3359_335970

theorem evaluate_expression : 7^3 - 4 * 7^2 + 6 * 7 - 1 = 188 := by
  sorry

end evaluate_expression_l3359_335970


namespace non_adjacent_white_balls_arrangements_select_balls_with_min_score_l3359_335912

/-- Represents the number of red balls in the bag -/
def num_red_balls : ℕ := 5

/-- Represents the number of white balls in the bag -/
def num_white_balls : ℕ := 4

/-- Represents the score for taking out a red ball -/
def red_ball_score : ℕ := 2

/-- Represents the score for taking out a white ball -/
def white_ball_score : ℕ := 1

/-- Represents the minimum required score -/
def min_score : ℕ := 8

/-- Represents the number of balls to be taken out -/
def balls_to_take : ℕ := 5

/-- Theorem for the number of ways to arrange balls with non-adjacent white balls -/
theorem non_adjacent_white_balls_arrangements : ℕ := by sorry

/-- Theorem for the number of ways to select balls with a minimum score -/
theorem select_balls_with_min_score : ℕ := by sorry

end non_adjacent_white_balls_arrangements_select_balls_with_min_score_l3359_335912


namespace five_solutions_l3359_335919

/-- The system of equations has exactly 5 distinct real solutions -/
theorem five_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ × ℝ)),
    Finset.card solutions = 5 ∧
    ∀ (x y z w : ℝ), (x, y, z, w) ∈ solutions ↔
      x = z + w + 2*z*w*x ∧
      y = w + x + w*x*y ∧
      z = x + y + x*y*z ∧
      w = y + z + 2*y*z*w := by
  sorry

end five_solutions_l3359_335919


namespace email_sample_not_representative_l3359_335939

/-- Represents the urban population -/
def UrbanPopulation : Type := Unit

/-- Represents a person in the urban population -/
def Person : Type := Unit

/-- Predicate for whether a person owns an email address -/
def has_email_address (p : Person) : Prop := sorry

/-- Predicate for whether a person uses the internet -/
def uses_internet (p : Person) : Prop := sorry

/-- Predicate for whether a person gets news from the internet -/
def gets_news_from_internet (p : Person) : Prop := sorry

/-- The sample of email address owners -/
def email_sample (n : ℕ) : Set Person := sorry

/-- A sample is representative if it accurately reflects the population characteristics -/
def is_representative (s : Set Person) : Prop := sorry

/-- Theorem stating that the email sample is not representative -/
theorem email_sample_not_representative (n : ℕ) : 
  ¬(is_representative (email_sample n)) := by sorry

end email_sample_not_representative_l3359_335939


namespace project_hours_difference_l3359_335940

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 135) 
  (pat kate mark : ℕ) 
  (h_pat_kate : pat = 2 * kate) 
  (h_pat_mark : pat * 3 = mark) 
  (h_sum : pat + kate + mark = total_hours) : 
  mark - kate = 75 := by
sorry

end project_hours_difference_l3359_335940


namespace sequence_general_term_l3359_335984

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →
  (a 1 = 1) →
  (∀ n : ℕ, (n + 1) * (a (n + 1))^2 - n * (a n)^2 + (a n) * (a (n + 1)) = 0) →
  (∀ n : ℕ, a n = 1 / n) :=
by sorry

end sequence_general_term_l3359_335984


namespace remaining_gasoline_l3359_335932

/-- Calculates the remaining gasoline in a car's tank after a journey -/
theorem remaining_gasoline
  (initial_gasoline : ℝ)
  (distance : ℝ)
  (fuel_consumption : ℝ)
  (h1 : initial_gasoline = 47)
  (h2 : distance = 275)
  (h3 : fuel_consumption = 12)
  : initial_gasoline - (distance * fuel_consumption / 100) = 14 := by
  sorry

end remaining_gasoline_l3359_335932


namespace quadratic_two_distinct_roots_specific_roots_l3359_335971

/-- The quadratic equation x^2 - (k+2)x + 2k - 1 = 0 has two distinct real roots for any real k -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (k+2)*x₁ + 2*k - 1 = 0 ∧ 
    x₂^2 - (k+2)*x₂ + 2*k - 1 = 0 :=
sorry

/-- When one root of the equation x^2 - (k+2)x + 2k - 1 = 0 is 3, k = 2 and the other root is 1 -/
theorem specific_roots : 
  ∃ k : ℝ, 3^2 - (k+2)*3 + 2*k - 1 = 0 ∧ 
    k = 2 ∧
    1^2 - (k+2)*1 + 2*k - 1 = 0 :=
sorry

end quadratic_two_distinct_roots_specific_roots_l3359_335971


namespace g_pi_third_equals_one_l3359_335999

/-- Given a function f and a constant w, φ, prove that g(π/3) = 1 -/
theorem g_pi_third_equals_one 
  (f : ℝ → ℝ) 
  (w φ : ℝ) 
  (h1 : ∀ x, f x = 5 * Real.cos (w * x + φ))
  (h2 : ∀ x, f (π/3 + x) = f (π/3 - x))
  (g : ℝ → ℝ)
  (h3 : ∀ x, g x = 4 * Real.sin (w * x + φ) + 1) :
  g (π/3) = 1 := by
sorry

end g_pi_third_equals_one_l3359_335999


namespace not_right_angled_triangle_l3359_335956

theorem not_right_angled_triangle : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 2 ∧ b = 3 ∧ c = 3) ∨
   (a = 1 ∧ b = 1 ∧ c = Real.sqrt 2)) ∧
  (a^2 + b^2 ≠ c^2) := by
sorry

end not_right_angled_triangle_l3359_335956


namespace linear_dependence_condition_l3359_335986

def vector1 : Fin 2 → ℝ := ![2, 3]
def vector2 (k : ℝ) : Fin 2 → ℝ := ![4, k]

def is_linearly_dependent (v1 v2 : Fin 2 → ℝ) : Prop :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ c1 • v1 + c2 • v2 = 0

theorem linear_dependence_condition (k : ℝ) :
  is_linearly_dependent vector1 (vector2 k) ↔ k = 6 := by
  sorry

end linear_dependence_condition_l3359_335986


namespace difference_of_variables_l3359_335923

theorem difference_of_variables (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by sorry

end difference_of_variables_l3359_335923


namespace stirling_number_second_kind_formula_l3359_335969

def stirling_number_second_kind (n r : ℕ) : ℚ :=
  (1 / r.factorial) *
    (Finset.sum (Finset.range (r + 1)) (fun k => 
      ((-1 : ℚ) ^ k * (r.choose k) * ((r - k) ^ n))))

theorem stirling_number_second_kind_formula (n r : ℕ) (h : n ≥ r) (hr : r > 0) :
  stirling_number_second_kind n r =
    (1 / r.factorial : ℚ) *
      (Finset.sum (Finset.range (r + 1)) (fun k => 
        ((-1 : ℚ) ^ k * (r.choose k) * ((r - k) ^ n)))) :=
by sorry

end stirling_number_second_kind_formula_l3359_335969


namespace rotation_result_l3359_335938

-- Define a type for the shapes
inductive Shape
  | Square
  | Pentagon
  | Ellipse

-- Define a type for the positions
inductive Position
  | X
  | Y
  | Z

-- Define a function to represent the initial configuration
def initial_config : Shape → Position
  | Shape.Square => Position.X
  | Shape.Pentagon => Position.Y
  | Shape.Ellipse => Position.Z

-- Define a function to represent the rotation
def rotate_180 (p : Position) : Position :=
  match p with
  | Position.X => Position.Y
  | Position.Y => Position.X
  | Position.Z => Position.Z

-- Theorem statement
theorem rotation_result :
  ∀ (s : Shape),
    rotate_180 (initial_config s) =
      match s with
      | Shape.Square => Position.Y
      | Shape.Pentagon => Position.X
      | Shape.Ellipse => Position.Z
  := by sorry

end rotation_result_l3359_335938


namespace smallest_k_for_error_bound_l3359_335958

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

theorem smallest_k_for_error_bound :
  ∃ (k : ℕ), (∀ (n : ℕ), n < k → |u n - L| > 1/2^1000) ∧
             |u k - L| ≤ 1/2^1000 ∧
             k = 9 := by
  sorry

end smallest_k_for_error_bound_l3359_335958


namespace greatest_product_base_seven_l3359_335968

/-- Represents a positive integer in base 7 --/
def BaseSeven := List Nat

/-- Converts a decimal number to base 7 --/
def toBaseSeven (n : Nat) : BaseSeven :=
  sorry

/-- Calculates the product of digits in a base 7 number --/
def productOfDigits (n : BaseSeven) : Nat :=
  sorry

/-- Theorem: The greatest possible product of digits in base 7 for numbers less than 2300 --/
theorem greatest_product_base_seven :
  (∃ (n : Nat), n < 2300 ∧
    (∀ (m : Nat), m < 2300 →
      productOfDigits (toBaseSeven m) ≤ productOfDigits (toBaseSeven n)) ∧
    productOfDigits (toBaseSeven n) = 1080) :=
  sorry

end greatest_product_base_seven_l3359_335968


namespace complex_polynomial_root_abs_d_l3359_335967

theorem complex_polynomial_root_abs_d (a b c d : ℤ) : 
  (a * (Complex.I + 3) ^ 5 + b * (Complex.I + 3) ^ 4 + c * (Complex.I + 3) ^ 3 + 
   d * (Complex.I + 3) ^ 2 + c * (Complex.I + 3) + b + a = 0) →
  (Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) →
  d.natAbs = 16 := by
sorry

end complex_polynomial_root_abs_d_l3359_335967


namespace xy_and_x_minus_y_squared_l3359_335989

theorem xy_and_x_minus_y_squared (x y : ℝ) 
  (sum_eq : x + y = 5) 
  (sum_squares_eq : x^2 + y^2 = 15) : 
  x * y = 5 ∧ (x - y)^2 = 5 := by sorry

end xy_and_x_minus_y_squared_l3359_335989


namespace symmetry_sum_l3359_335931

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_origin (a, -3) (4, b) → a + b = -1 := by
  sorry

end symmetry_sum_l3359_335931


namespace smallest_n_greater_than_threshold_l3359_335979

/-- The first term of the arithmetic sequence -/
def a₁ : ℕ := 11

/-- The common difference of the arithmetic sequence -/
def d : ℕ := 6

/-- The threshold value -/
def threshold : ℕ := 2017

/-- The n-th term of the arithmetic sequence -/
def aₙ (n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- The proposition to be proved -/
theorem smallest_n_greater_than_threshold :
  (∀ k ≥ 336, aₙ k > threshold) ∧
  (∀ m < 336, ∃ l ≥ m, aₙ l ≤ threshold) :=
sorry

end smallest_n_greater_than_threshold_l3359_335979


namespace sqrt_equation_solution_l3359_335998

theorem sqrt_equation_solution :
  ∀ x : ℝ, x ≥ 0 → x + 4 ≥ 0 → Real.sqrt x + Real.sqrt (x + 4) = 12 → x = 1225 / 36 := by
  sorry

end sqrt_equation_solution_l3359_335998


namespace factorization_cubic_minus_xy_squared_l3359_335995

theorem factorization_cubic_minus_xy_squared (x y : ℝ) : 
  x^3 - x*y^2 = x*(x + y)*(x - y) := by sorry

end factorization_cubic_minus_xy_squared_l3359_335995


namespace alpha_gamma_relation_l3359_335976

theorem alpha_gamma_relation (α β γ : ℝ) 
  (h1 : β = 10^(1 / (1 - Real.log α)))
  (h2 : γ = 10^(1 / (1 - Real.log β))) :
  α = 10^(1 / (1 - Real.log γ)) := by
  sorry

end alpha_gamma_relation_l3359_335976


namespace inequalities_proof_l3359_335914

theorem inequalities_proof (a b : ℝ) (h : a > b) (h0 : b > 0) :
  (Real.sqrt a > Real.sqrt b) ∧ (a - 1/a > b - 1/b) := by
  sorry

end inequalities_proof_l3359_335914


namespace max_weight_difference_is_0_6_l3359_335975

/-- Represents the weight range of a flour bag -/
structure FlourBag where
  center : ℝ
  tolerance : ℝ

/-- Calculates the maximum weight of a flour bag -/
def max_weight (bag : FlourBag) : ℝ := bag.center + bag.tolerance

/-- Calculates the minimum weight of a flour bag -/
def min_weight (bag : FlourBag) : ℝ := bag.center - bag.tolerance

/-- Theorem: The maximum difference in weights between any two bags is 0.6 kg -/
theorem max_weight_difference_is_0_6 (bag1 bag2 bag3 : FlourBag)
  (h1 : bag1 = ⟨25, 0.1⟩)
  (h2 : bag2 = ⟨25, 0.2⟩)
  (h3 : bag3 = ⟨25, 0.3⟩) :
  (max_weight bag3 - min_weight bag3) = 0.6 :=
by sorry

end max_weight_difference_is_0_6_l3359_335975


namespace johns_labor_cost_johns_specific_labor_cost_l3359_335959

/-- Represents the problem of calculating labor costs for John's table-making business --/
theorem johns_labor_cost (trees : ℕ) (planks_per_tree : ℕ) (planks_per_table : ℕ) 
  (price_per_table : ℕ) (total_profit : ℕ) : ℕ :=
  let total_planks := trees * planks_per_tree
  let total_tables := total_planks / planks_per_table
  let total_revenue := total_tables * price_per_table
  let labor_cost := total_revenue - total_profit
  labor_cost

/-- The specific instance of John's labor cost calculation --/
theorem johns_specific_labor_cost : 
  johns_labor_cost 30 25 15 300 12000 = 3000 := by
  sorry

end johns_labor_cost_johns_specific_labor_cost_l3359_335959


namespace shifted_function_eq_minus_three_x_minus_four_l3359_335904

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Shifts a linear function vertically by a given amount -/
def shift_vertical (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + shift }

/-- The original linear function y = -3x -/
def original_function : LinearFunction :=
  { m := -3, b := 0 }

/-- The amount to shift the function down -/
def shift_amount : ℝ := -4

theorem shifted_function_eq_minus_three_x_minus_four :
  shift_vertical original_function shift_amount = { m := -3, b := -4 } := by
  sorry

end shifted_function_eq_minus_three_x_minus_four_l3359_335904


namespace expand_quadratic_l3359_335994

theorem expand_quadratic (x : ℝ) : (2*x + 3)*(4*x - 9) = 8*x^2 - 6*x - 27 := by
  sorry

end expand_quadratic_l3359_335994


namespace race_probability_l3359_335991

theorem race_probability (pX pY pZ : ℚ) : 
  pX = 1/4 → pY = 1/12 → pZ = 1/7 → 
  (pX + pY + pZ : ℚ) = 10/21 := by sorry

end race_probability_l3359_335991


namespace difference_x_y_l3359_335992

theorem difference_x_y : ∀ (x y : ℤ), x + y = 250 → y = 225 → |x - y| = 200 := by
  sorry

end difference_x_y_l3359_335992


namespace linear_equation_condition_l3359_335978

theorem linear_equation_condition (m : ℤ) : 
  (∃ a b : ℝ, ∀ x : ℝ, (m + 1 : ℝ) * x^(|m|) + 3 = a * x + b) ↔ m = 1 :=
sorry

end linear_equation_condition_l3359_335978


namespace isosceles_triangle_condition_l3359_335977

theorem isosceles_triangle_condition (A B C : Real) :
  (A > 0) → (B > 0) → (C > 0) → (A + B + C = π) →
  (Real.log (Real.sin A) - Real.log (Real.cos B) - Real.log (Real.sin C) = Real.log 2) →
  ∃ (x y : Real), (x = y) ∧ 
  ((A = x ∧ B = y ∧ C = y) ∨ (A = y ∧ B = x ∧ C = y) ∨ (A = y ∧ B = y ∧ C = x)) :=
by sorry

end isosceles_triangle_condition_l3359_335977


namespace optimal_vegetable_transport_plan_l3359_335945

/-- Represents the capacity and rental cost of a truck type -/
structure TruckType where
  capacity : ℕ
  rentalCost : ℕ

/-- The problem setup -/
def vegetableTransportProblem (typeA typeB : TruckType) : Prop :=
  -- Conditions
  2 * typeA.capacity + typeB.capacity = 10 ∧
  typeA.capacity + 2 * typeB.capacity = 11 ∧
  -- Define a function to calculate the total capacity of a plan
  (λ (x y : ℕ) => x * typeA.capacity + y * typeB.capacity) = 
    (λ (x y : ℕ) => 31) ∧
  -- Define a function to calculate the total cost of a plan
  (λ (x y : ℕ) => x * typeA.rentalCost + y * typeB.rentalCost) = 
    (λ (x y : ℕ) => 940) ∧
  -- The optimal plan
  (1 : ℕ) * typeA.capacity + (7 : ℕ) * typeB.capacity = 31

/-- The theorem to prove -/
theorem optimal_vegetable_transport_plan :
  ∃ (typeA typeB : TruckType),
    vegetableTransportProblem typeA typeB ∧
    typeA.rentalCost = 100 ∧
    typeB.rentalCost = 120 :=
  sorry


end optimal_vegetable_transport_plan_l3359_335945


namespace gardner_cupcakes_l3359_335941

/-- The number of cupcakes baked by Mr. Gardner -/
def cupcakes : ℕ := sorry

/-- The number of cookies baked -/
def cookies : ℕ := 20

/-- The number of brownies baked -/
def brownies : ℕ := 35

/-- The number of students in the class -/
def students : ℕ := 20

/-- The number of sweet treats each student receives -/
def treats_per_student : ℕ := 4

/-- The total number of sweet treats -/
def total_treats : ℕ := students * treats_per_student

theorem gardner_cupcakes : cupcakes = 25 := by
  sorry

end gardner_cupcakes_l3359_335941


namespace table_length_proof_l3359_335925

theorem table_length_proof (table_width : ℕ) (sheet_width sheet_height : ℕ) :
  table_width = 80 ∧ 
  sheet_width = 8 ∧ 
  sheet_height = 5 ∧ 
  (∃ n : ℕ, table_width = sheet_width + n ∧ n + 1 = sheet_width - sheet_height + 1) →
  ∃ table_length : ℕ, table_length = 77 ∧ table_length = table_width - (sheet_width - sheet_height) :=
by sorry

end table_length_proof_l3359_335925


namespace course_selection_theorem_l3359_335949

def type_a_courses : ℕ := 3
def type_b_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

/-- The number of ways to select courses from two types of electives -/
def number_of_selections : ℕ :=
  Nat.choose type_a_courses 1 * Nat.choose type_b_courses 2 +
  Nat.choose type_a_courses 2 * Nat.choose type_b_courses 1

theorem course_selection_theorem :
  number_of_selections = 30 := by sorry

end course_selection_theorem_l3359_335949


namespace min_xy_value_l3359_335910

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x * y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = x₀*y₀ ∧ x₀ * y₀ = 8 :=
by sorry

end min_xy_value_l3359_335910


namespace election_winner_percentage_l3359_335916

theorem election_winner_percentage (total_votes : ℕ) (vote_majority : ℕ) : 
  total_votes = 600 → vote_majority = 240 → 
  (70 : ℚ) / 100 * total_votes = (total_votes + vote_majority) / 2 := by
  sorry

end election_winner_percentage_l3359_335916


namespace sampling_methods_correct_l3359_335965

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a sampling scenario -/
structure SamplingScenario where
  total_items : ℕ
  sample_size : ℕ
  is_homogeneous : Bool
  has_structure : Bool
  has_strata : Bool

/-- Determines the most appropriate sampling method for a given scenario -/
def appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.is_homogeneous then SamplingMethod.SimpleRandom
  else if scenario.has_structure then SamplingMethod.Systematic
  else if scenario.has_strata then SamplingMethod.Stratified
  else SamplingMethod.SimpleRandom

theorem sampling_methods_correct :
  (appropriate_sampling_method ⟨10, 3, true, false, false⟩ = SamplingMethod.SimpleRandom) ∧
  (appropriate_sampling_method ⟨1280, 32, false, true, false⟩ = SamplingMethod.Systematic) ∧
  (appropriate_sampling_method ⟨12, 50, false, false, true⟩ = SamplingMethod.Stratified) :=
by sorry

end sampling_methods_correct_l3359_335965


namespace dog_adoptions_l3359_335951

theorem dog_adoptions (dog_fee cat_fee : ℕ) (cat_adoptions : ℕ) (donation_fraction : ℚ) (donation_amount : ℕ) : 
  dog_fee = 15 →
  cat_fee = 13 →
  cat_adoptions = 3 →
  donation_fraction = 1/3 →
  donation_amount = 53 →
  ∃ (dog_adoptions : ℕ), 
    dog_adoptions = 8 ∧ 
    (↑donation_amount : ℚ) = donation_fraction * (↑dog_fee * ↑dog_adoptions + ↑cat_fee * ↑cat_adoptions) :=
by sorry

end dog_adoptions_l3359_335951


namespace min_value_cubic_function_l3359_335948

theorem min_value_cubic_function (y : ℝ) (h : y > 0) :
  y^2 + 10*y + 100/y^3 ≥ 50^(2/3) + 10 * 50^(1/3) + 2 ∧
  (y^2 + 10*y + 100/y^3 = 50^(2/3) + 10 * 50^(1/3) + 2 ↔ y = 50^(1/3)) :=
by sorry

end min_value_cubic_function_l3359_335948


namespace isosceles_triangle_vertex_angle_l3359_335988

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ α = γ) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 70°
  (α = 70 ∨ β = 70 ∨ γ = 70) →
  -- The vertex angle (the one that's not equal to the other two) is either 70° or 40°
  (((α ≠ β ∧ α ≠ γ) → α = 70 ∨ α = 40) ∧
   ((β ≠ α ∧ β ≠ γ) → β = 70 ∨ β = 40) ∧
   ((γ ≠ α ∧ γ ≠ β) → γ = 70 ∨ γ = 40)) :=
by sorry

end isosceles_triangle_vertex_angle_l3359_335988


namespace m_range_l3359_335924

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, 
  m * x₁^2 - x₁ + m - 4 = 0 ∧ 
  m * x₂^2 - x₂ + m - 4 = 0 ∧ 
  x₁ > 0 ∧ x₂ < 0

-- Define the theorem
theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) → ¬(p m) → m ≥ 1 + Real.sqrt 2 ∧ m < 4 :=
sorry

end m_range_l3359_335924


namespace range_of_z_l3359_335953

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  let z := x^2 + 4*y^2
  4 ≤ z ∧ z ≤ 12 := by
sorry

end range_of_z_l3359_335953


namespace larger_number_proof_l3359_335955

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 24672)
  (h2 : L = 13 * S + 257) :
  L = 26706 := by
  sorry

end larger_number_proof_l3359_335955


namespace park_area_l3359_335907

/-- The area of a rectangular park with a modified perimeter -/
theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w + 5 = 80) (h2 : l = 3 * w) :
  l * w = 263.6719 := by
  sorry

end park_area_l3359_335907


namespace reflection_of_P_across_x_axis_l3359_335901

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 5)

theorem reflection_of_P_across_x_axis :
  reflect_x P = (-2, -5) := by sorry

end reflection_of_P_across_x_axis_l3359_335901


namespace prob_same_outcome_equals_half_l3359_335908

-- Define the success probabilities for two independent events
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.8

-- Define the probability of both events resulting in the same outcome
def prob_same_outcome : ℝ := (prob_A * prob_B) + ((1 - prob_A) * (1 - prob_B))

-- Theorem statement
theorem prob_same_outcome_equals_half : prob_same_outcome = 0.5 := by
  sorry

end prob_same_outcome_equals_half_l3359_335908


namespace polynomial_range_l3359_335909

theorem polynomial_range (x : ℝ) :
  x^2 - 7*x + 12 < 0 →
  90 < x^3 + 5*x^2 + 6*x ∧ x^3 + 5*x^2 + 6*x < 168 := by
sorry

end polynomial_range_l3359_335909


namespace farmer_reward_distribution_l3359_335935

theorem farmer_reward_distribution (total_farmers : ℕ) (total_budget : ℕ) 
  (self_employed_reward : ℕ) (stable_employment_reward : ℕ) 
  (h1 : total_farmers = 60)
  (h2 : total_budget = 100000)
  (h3 : self_employed_reward = 1000)
  (h4 : stable_employment_reward = 2000) :
  ∃ (self_employed : ℕ) (stable_employment : ℕ),
    self_employed + stable_employment = total_farmers ∧
    self_employed * self_employed_reward + 
    stable_employment * (self_employed_reward + stable_employment_reward) = total_budget ∧
    self_employed = 40 ∧ 
    stable_employment = 20 := by
  sorry

end farmer_reward_distribution_l3359_335935


namespace bmw_sales_count_l3359_335987

def total_cars : ℕ := 250
def mercedes_percent : ℚ := 18 / 100
def toyota_percent : ℚ := 25 / 100
def acura_percent : ℚ := 15 / 100

theorem bmw_sales_count :
  (total_cars : ℚ) * (1 - (mercedes_percent + toyota_percent + acura_percent)) = 105 := by
  sorry

end bmw_sales_count_l3359_335987


namespace dvd_packs_cost_l3359_335983

/-- Proves that given the cost of each DVD pack and the number of packs that can be bought,
    the total amount of money available is correct. -/
theorem dvd_packs_cost (cost_per_pack : ℕ) (num_packs : ℕ) (total_money : ℕ) :
  cost_per_pack = 26 → num_packs = 4 → total_money = cost_per_pack * num_packs →
  total_money = 104 := by
  sorry

end dvd_packs_cost_l3359_335983


namespace line_equation_sum_l3359_335963

/-- Given two points on a line, proves that m + b = 7 where y = mx + b is the equation of the line. -/
theorem line_equation_sum (x₁ y₁ x₂ y₂ m b : ℚ) : 
  x₁ = 1 → y₁ = 7 → x₂ = -2 → y₂ = -1 →
  m = (y₂ - y₁) / (x₂ - x₁) →
  y₁ = m * x₁ + b →
  m + b = 7 := by
sorry

end line_equation_sum_l3359_335963


namespace f_root_and_positivity_l3359_335985

noncomputable def f (x : ℝ) : ℝ := 2^x - 2/x

theorem f_root_and_positivity :
  (∃! x : ℝ, f x = 0 ∧ x = 1) ∧
  (∀ x : ℝ, x ≠ 0 → (f x > 0 ↔ x < 0 ∨ x > 1)) :=
by sorry

end f_root_and_positivity_l3359_335985


namespace eldest_child_age_l3359_335981

theorem eldest_child_age 
  (n : ℕ) 
  (d : ℕ) 
  (sum : ℕ) 
  (h1 : n = 5) 
  (h2 : d = 2) 
  (h3 : sum = 50) : 
  (sum - (n * (n - 1) / 2) * d) / n + (n - 1) * d = 14 := by
  sorry

end eldest_child_age_l3359_335981


namespace inequality_proof_l3359_335933

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
sorry

end inequality_proof_l3359_335933


namespace basketball_shots_l3359_335957

theorem basketball_shots (total_points : ℕ) (total_shots : ℕ) 
  (h_points : total_points = 26) (h_shots : total_shots = 11) :
  ∃ (three_pointers two_pointers : ℕ),
    three_pointers + two_pointers = total_shots ∧
    3 * three_pointers + 2 * two_pointers = total_points ∧
    three_pointers = 4 := by
  sorry

end basketball_shots_l3359_335957


namespace total_meals_sold_l3359_335966

/-- Given the ratio of kids meals to adult meals to seniors' meals and the number of kids meals sold,
    calculate the total number of meals sold. -/
theorem total_meals_sold (kids_ratio adult_ratio seniors_ratio kids_meals : ℕ) : 
  kids_ratio > 0 → 
  adult_ratio > 0 → 
  seniors_ratio > 0 → 
  kids_ratio = 3 → 
  adult_ratio = 2 → 
  seniors_ratio = 1 → 
  kids_meals = 12 → 
  kids_meals + (adult_ratio * kids_meals / kids_ratio) + (seniors_ratio * kids_meals / kids_ratio) = 24 := by
sorry

end total_meals_sold_l3359_335966


namespace car_wash_earnings_l3359_335936

theorem car_wash_earnings (total : ℕ) (lisa : ℕ) (tommy : ℕ) : 
  total = 60 → 
  lisa = total / 2 → 
  tommy = lisa / 2 → 
  lisa - tommy = 15 := by
sorry

end car_wash_earnings_l3359_335936


namespace sequence_a_10_l3359_335915

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, a (p + q) = a p * a q)

theorem sequence_a_10 (a : ℕ → ℝ) 
  (h_prop : sequence_property a) 
  (h_a8 : a 8 = 16) : 
  a 10 = 32 := by
sorry

end sequence_a_10_l3359_335915


namespace obtuse_angle_range_l3359_335996

def vector_a : ℝ × ℝ := (2, -1)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 3)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def is_obtuse (v w : ℝ × ℝ) : Prop := dot_product v w < 0

theorem obtuse_angle_range (t : ℝ) :
  is_obtuse vector_a (vector_b t) →
  t ∈ (Set.Iio (-6) ∪ Set.Ioo (-6) (3/2)) :=
sorry

end obtuse_angle_range_l3359_335996


namespace intersection_of_P_and_Q_l3359_335927

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = x}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≤ 2} := by sorry

end intersection_of_P_and_Q_l3359_335927


namespace longest_side_length_l3359_335922

-- Define a triangle with angle ratio 1:2:3 and shortest side 5 cm
structure SpecialTriangle where
  -- a, b, c are the side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angle A is opposite to side a, B to b, C to c
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  -- Conditions
  angle_ratio : angleA / angleB = 1/2 ∧ angleB / angleC = 2/3
  shortest_side : min a (min b c) = 5
  -- Triangle properties
  sum_angles : angleA + angleB + angleC = π
  -- Law of sines
  law_of_sines : a / (Real.sin angleA) = b / (Real.sin angleB)
                 ∧ b / (Real.sin angleB) = c / (Real.sin angleC)

-- Theorem statement
theorem longest_side_length (t : SpecialTriangle) : max t.a (max t.b t.c) = 10 := by
  sorry

end longest_side_length_l3359_335922


namespace arithmetic_sequence_common_difference_l3359_335903

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h1 : seq.S 5 = -15)
  (h2 : seq.a 2 + seq.a 5 = -2) :
  seq.d = 4 := by
sorry

end arithmetic_sequence_common_difference_l3359_335903


namespace calculation_equality_l3359_335952

theorem calculation_equality : 
  |3 - Real.sqrt 12| + (1/3)⁻¹ - 4 * Real.sin (60 * π / 180) + (Real.sqrt 2)^2 = 2 := by sorry

end calculation_equality_l3359_335952


namespace ammonium_iodide_required_l3359_335950

-- Define the molecules and their molar quantities
structure Reaction where
  nh4i : ℝ  -- Ammonium iodide
  koh : ℝ   -- Potassium hydroxide
  nh3 : ℝ   -- Ammonia
  ki : ℝ    -- Potassium iodide
  h2o : ℝ   -- Water

-- Define the balanced chemical equation
def balanced_equation (r : Reaction) : Prop :=
  r.nh4i = r.koh ∧ r.nh4i = r.nh3 ∧ r.nh4i = r.ki ∧ r.nh4i = r.h2o

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.koh = 3 ∧ r.nh3 = 3 ∧ r.ki = 3 ∧ r.h2o = 3

-- Theorem statement
theorem ammonium_iodide_required (r : Reaction) 
  (h1 : balanced_equation r) (h2 : given_conditions r) : 
  r.nh4i = 3 :=
sorry

end ammonium_iodide_required_l3359_335950


namespace unique_zero_composition_implies_m_bound_l3359_335918

/-- Given a function f(x) = x^2 + 2x + m where m is a real number,
    if f(f(x)) has exactly one zero, then 0 < m < 1 -/
theorem unique_zero_composition_implies_m_bound 
  (m : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + 2*x + m)
  (h2 : ∃! x, f (f x) = 0) :
  0 < m ∧ m < 1 := by
  sorry

end unique_zero_composition_implies_m_bound_l3359_335918


namespace smallest_sum_reciprocals_l3359_335997

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ (a : ℕ) + (b : ℕ) = 49 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → (c : ℕ) + (d : ℕ) ≥ 49 :=
by sorry

end smallest_sum_reciprocals_l3359_335997


namespace max_value_of_a_l3359_335911

theorem max_value_of_a (a b c d : ℕ) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ), a' = 8924 ∧ a' < 3 * b' ∧ b' < 4 * c' ∧ c' < 5 * d' ∧ d' < 150 :=
by sorry

end max_value_of_a_l3359_335911


namespace continuity_at_four_l3359_335902

/-- Continuity of f(x) = -2x^2 + 9 at x₀ = 4 -/
theorem continuity_at_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |(-2 * x^2 + 9) - (-2 * 4^2 + 9)| < ε := by
sorry

end continuity_at_four_l3359_335902


namespace coin_difference_is_eight_l3359_335928

/-- Represents the available coin denominations in cents -/
def coin_denominations : List Nat := [5, 10, 20, 25]

/-- The amount to be paid in cents -/
def amount_to_pay : Nat := 50

/-- Calculates the minimum number of coins needed to make the given amount -/
def min_coins (amount : Nat) (denominations : List Nat) : Nat :=
  sorry

/-- Calculates the maximum number of coins needed to make the given amount -/
def max_coins (amount : Nat) (denominations : List Nat) : Nat :=
  sorry

/-- Proves that the difference between the maximum and minimum number of coins
    needed to make 50 cents using the given denominations is 8 -/
theorem coin_difference_is_eight :
  max_coins amount_to_pay coin_denominations - min_coins amount_to_pay coin_denominations = 8 :=
by sorry

end coin_difference_is_eight_l3359_335928


namespace consecutive_palindrome_diff_l3359_335944

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The set of all five-digit palindromes -/
def five_digit_palindromes : Set ℕ :=
  {n : ℕ | is_five_digit_palindrome n}

/-- The theorem stating the possible differences between consecutive five-digit palindromes -/
theorem consecutive_palindrome_diff 
  (a b : ℕ) 
  (ha : a ∈ five_digit_palindromes) 
  (hb : b ∈ five_digit_palindromes)
  (hless : a < b)
  (hconsec : ∀ x, x ∈ five_digit_palindromes → a < x → x < b → False) :
  b - a = 100 ∨ b - a = 110 ∨ b - a = 11 :=
sorry

end consecutive_palindrome_diff_l3359_335944


namespace dog_speed_l3359_335926

/-- Proves that a dog catching a rabbit with given parameters runs at 24 miles per hour -/
theorem dog_speed (rabbit_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  rabbit_speed = 15 →
  head_start = 0.6 →
  catch_up_time = 4 / 60 →
  let dog_distance := rabbit_speed * catch_up_time + head_start
  dog_distance / catch_up_time = 24 := by sorry

end dog_speed_l3359_335926


namespace positive_numbers_inequalities_l3359_335980

theorem positive_numbers_inequalities (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≥ 1 ∧
  a^2/(b+c) + b^2/(a+c) + c^2/(a+b) ≥ 1/2 := by
sorry

end positive_numbers_inequalities_l3359_335980


namespace conditional_statement_else_branch_l3359_335934

/-- Represents a conditional statement structure -/
inductive ConditionalStatement
  | ifThenElse (condition : Prop) (thenBranch : Prop) (elseBranch : Prop)

/-- Represents the execution of a conditional statement -/
def executeConditional (stmt : ConditionalStatement) (conditionMet : Bool) : Prop :=
  match stmt with
  | ConditionalStatement.ifThenElse _ thenBranch elseBranch => 
      if conditionMet then thenBranch else elseBranch

theorem conditional_statement_else_branch 
  (stmt : ConditionalStatement) (conditionMet : Bool) :
  ¬conditionMet → 
  executeConditional stmt conditionMet = 
    match stmt with
    | ConditionalStatement.ifThenElse _ _ elseBranch => elseBranch :=
by
  sorry

end conditional_statement_else_branch_l3359_335934


namespace geometric_sequence_sum_l3359_335982

theorem geometric_sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * r) →
  a 3 = 256 →
  a 5 = 4 →
  a 3 + a 4 = 80 :=
by
  sorry

end geometric_sequence_sum_l3359_335982


namespace change_calculation_l3359_335942

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def bills_given : ℕ := 20 * 2
def coins_given : ℕ := 3

def total_cost : ℕ := flour_cost + cake_stand_cost
def total_paid : ℕ := bills_given + coins_given

theorem change_calculation (change : ℕ) : 
  change = total_paid - total_cost := by sorry

end change_calculation_l3359_335942


namespace fraction_relation_l3359_335964

theorem fraction_relation (p r s u : ℝ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 := by
sorry

end fraction_relation_l3359_335964


namespace trailing_zeros_theorem_l3359_335920

/-- Count trailing zeros in factorial -/
def trailingZeros (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125)

/-- Check if n satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  ∃ k : ℕ, trailingZeros (n + 3) = k ∧ trailingZeros (2 * n + 6) = 4 * k

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem -/
theorem trailing_zeros_theorem :
  ∃ t : ℕ,
    (∃ a b c d : ℕ,
      a > 6 ∧ b > 6 ∧ c > 6 ∧ d > 6 ∧
      a < b ∧ b < c ∧ c < d ∧
      satisfiesCondition a ∧ satisfiesCondition b ∧ satisfiesCondition c ∧ satisfiesCondition d ∧
      t = a + b + c + d ∧
      ∀ n : ℕ, n > 6 ∧ satisfiesCondition n → n ≥ a) ∧
    sumOfDigits t = 4 :=
  sorry

end trailing_zeros_theorem_l3359_335920


namespace arithmetic_sequence_problem_l3359_335900

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence {a_n} where a_4 = 5 and a_9 = 17, a_14 = 29 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_a4 : a 4 = 5) 
    (h_a9 : a 9 = 17) : 
  a 14 = 29 := by
sorry

end arithmetic_sequence_problem_l3359_335900


namespace unique_factors_of_135135_l3359_335943

theorem unique_factors_of_135135 :
  ∃! (a b c d e f : ℕ),
    1 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
    a * b * c * d * e * f = 135135 ∧
    a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9 ∧ e = 11 ∧ f = 13 :=
by sorry

end unique_factors_of_135135_l3359_335943


namespace triangle_probability_l3359_335921

def stick_lengths : List ℕ := [1, 4, 6, 8, 9, 10, 12, 15]

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_perimeter_gt_20 (a b c : ℕ) : Prop :=
  a + b + c > 20

def valid_triangle_count : ℕ := 16

def total_combinations : ℕ := Nat.choose 8 3

theorem triangle_probability : 
  (valid_triangle_count : ℚ) / total_combinations = 2 / 7 := by sorry

end triangle_probability_l3359_335921


namespace no_unchanged_sum_l3359_335930

theorem no_unchanged_sum : ¬∃ (A B : ℕ), A + B = 2022 ∧ A / 2 + 3 * B = A + B := by
  sorry

end no_unchanged_sum_l3359_335930


namespace discount_restoration_l3359_335962

theorem discount_restoration (original_price : ℝ) (discount_rate : ℝ) (restoration_rate : ℝ) : 
  discount_rate = 0.2 ∧ restoration_rate = 0.25 →
  original_price * (1 - discount_rate) * (1 + restoration_rate) = original_price :=
by sorry

end discount_restoration_l3359_335962


namespace exponential_function_implies_a_eq_three_l3359_335973

/-- A function f: ℝ → ℝ is exponential if there exist constants b > 0, b ≠ 1, and c such that f(x) = c * b^x for all x ∈ ℝ. -/
def IsExponentialFunction (f : ℝ → ℝ) : Prop :=
  ∃ (b c : ℝ), b > 0 ∧ b ≠ 1 ∧ ∀ x, f x = c * b^x

/-- If f(x) = (a-2) * a^x is an exponential function, then a = 3. -/
theorem exponential_function_implies_a_eq_three (a : ℝ) :
  IsExponentialFunction (fun x ↦ (a - 2) * a^x) → a = 3 := by
  sorry

end exponential_function_implies_a_eq_three_l3359_335973
